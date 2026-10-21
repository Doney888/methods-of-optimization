// AWT и ImageIO рисуют PNG; TreeMap хранит пробы, PriorityQueue выбирает интервал.
// Готовых оптимизаторов здесь нет: шаги метода реализованы ниже.
import java.awt.BasicStroke
import java.awt.Color
import java.awt.Font
import java.awt.RenderingHints
import java.awt.image.BufferedImage
import java.io.File
import java.util.Locale
import java.util.PriorityQueue
import java.util.TreeMap
import javax.imageio.ImageIO
import kotlin.math.*

// Узел дерева: число, переменная, операция или вызов функции.
// kind задаёт тип узла, text — число, имя или знак операции.
// children — подвыражения: у суммы их два, у cos один, у числа нет.
data class Expr(val kind: String, val text: String = "", val children: List<Expr> = emptyList())
// Границы значений выражения и оценка его константы Липшица.
// lo <= выражение <= hi на всём [a,b]; lip ограничивает скорость изменения.
// Это границы самого подвыражения, не итоговые оценки глобального минимума.
data class Bound(val lo: Double, val hi: Double, val lip: Double)

// Разбираем строку с учётом скобок и приоритетов операций.
class ExpressionParser(expression: String) {
    // Исходную строку сохраняем один раз; pos — индекс следующего символа.
    private val source = expression
    // var позволяет двигать позицию при чтении; source объявлена val и не переназначается.
    private var pos = 0
    // Пробелы между частями формулы не влияют на её смысл.
    private fun skip() { while (pos < source.length && source[pos].isWhitespace()) pos++ }
    // Если нужный знак стоит в текущей позиции, пропускаем его и возвращаем true.
    // Если его нет, оставляем позицию на месте: вызывающий код пробует другой вариант.
    private fun take(token: String): Boolean {
        skip()
        if (!source.startsWith(token, pos)) return false
        pos += token.length
        return true
    }
    fun parse(): Expr {
        // Начинаем с самого низкого приоритета; более сильные операции читаются глубже.
        val node = sum()
        skip()
        // Вся строка должна быть разобрана: незнакомый остаток не игнорируем.
        require(pos == source.length) { "Лишний символ в позиции $pos" }
        return node
    }
    // Сначала разбираем произведения, затем сложение и вычитание.
    private fun sum(): Expr {
        // Первый операнд может содержать умножения, степени и скобки.
        var node = product()
        while (true) node = when {
            // Уже собранное дерево становится левым операндом новой операции.
            take("+") -> Expr("binary", "+", listOf(node, product()))
            take("-") -> Expr("binary", "-", listOf(node, product()))
            else -> return node
        }
    }
    // Этот уровень связывает умножения и деления до перехода к суммам.
    private fun product(): Expr {
        var node = unary()
        while (true) {
            skip()
            node = when {
                // Два символа ** относятся к степени, а не к одиночному умножению.
                source.startsWith("**", pos) -> return node
                take("*") -> Expr("binary", "*", listOf(node, unary()))
                take("/") -> Expr("binary", "/", listOf(node, unary()))
                else -> return node
            }
        }
    }
    // Унарный минус меняет знак одного выражения; это не вычитание двух операндов.
    // Степень имеет приоритет над ним: -x**2 читается как -(x**2).
    private fun unary(): Expr = when {
        take("+") -> Expr("unary", "+", listOf(unary()))
        take("-") -> Expr("unary", "-", listOf(unary()))
        else -> power()
    }
    // Степени разбираем справа налево: 2**3**2 = 2**(3**2).
    private fun power(): Expr {
        val node = atom()
        // Рекурсивный вызов справа разрешает и отрицательный показатель: x**-2.
        return if (take("**")) Expr("binary", "**", listOf(node, unary())) else node
    }
    // Читаем отдельное число, имя, функцию или выражение в скобках.
    private fun atom(): Expr {
        skip()
        // Внутри скобок снова разрешено полное выражение; закрывающая скобка обязательна.
        if (take("(")) {
            val node = sum()
            require(take(")")) { "Нет закрывающей скобки" }
            return node
        }
        require(pos < source.length) { "Неожиданный конец выражения" }
        if (source[pos].isDigit() || source[pos] == '.') {
            // Принимаем целые, дробные и научную запись вроде 1.5e-3.
            // Ведущий знак обрабатывается отдельно в unary.
            val pattern = Regex("(?:[0-9]+(?:\\.[0-9]*)?|\\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
            val match = pattern.find(source, pos)
            require(match != null && match.range.first == pos) { "Неверное число" }
            // Продвигаем позицию ровно за найденное число.
            pos = match.range.last + 1
            require(match.value.toDouble().isFinite()) { "Неконечная константа" }
            return Expr("number", match.value)
        }
        // Читаем буквенное имя целиком, затем сверяем с разрешённым набором.
        val start = pos
        while (pos < source.length && source[pos].isLetter()) pos++
        require(pos > start) { "Недопустимый символ в позиции $pos" }
        val name = source.substring(start, pos)
        if (name in setOf("x", "pi", "e")) return Expr("name", name)
        require(name in setOf("sin", "cos", "exp", "log", "sqrt", "abs") && take("(")) {
            "Неизвестное имя или вызов: $name"
        }
        // Аргумент функции тоже может быть сложным выражением: cos(2*pi*x).
        val argument = sum()
        require(take(")")) { "Нет закрывающей скобки функции" }
        return Expr("call", name, listOf(argument))
    }
}

// Вычисляем дерево выражения при заданном x.
fun evaluate(node: Expr, x: Double): Double {
    // Сначала рекурсивно считаем дочерние узлы, затем выполняем текущую операцию.
    // Для листа список args пуст; для бинарной операции args содержит два числа.
    val args = node.children.map { evaluate(it, x) }
    return when (node.kind) {
        "number" -> node.text.toDouble()
        // x берём из параметра вызова, pi и e — математические константы.
        "name" -> when (node.text) { "x" -> x; "pi" -> PI; else -> E }
        "unary" -> if (node.text == "-") -args[0] else args[0]
        "call" -> when (node.text) {
            "sin" -> sin(args[0]); "cos" -> cos(args[0]); "exp" -> exp(args[0])
            "log" -> ln(args[0]); "sqrt" -> sqrt(args[0]); else -> abs(args[0])
        }
        else -> when (node.text) {
            "+" -> args[0] + args[1]; "-" -> args[0] - args[1]
            "*" -> args[0] * args[1]; "/" -> args[0] / args[1]; else -> args[0].pow(args[1])
        }
    }
}

fun expressionBound(node: Expr, a: Double, b: Double): Bound {
    // Диапазон выражения и аналитическая верхняя оценка его константы Липшица.
    // Константа не меняется: её диапазон — одно число, а L = 0.
    if (node.kind == "number") return Bound(node.text.toDouble(), node.text.toDouble(), 0.0)
    if (node.kind == "name") return when (node.text) {
        // Для x на [a,b] диапазон известен непосредственно, L(x) = 1.
        "x" -> Bound(a, b, 1.0); "pi" -> Bound(PI, PI, 0.0); else -> Bound(E, E, 0.0)
    }
    // Рекурсивно получаем диапазон и L левого операнда или аргумента функции.
    val u = expressionBound(node.children[0], a, b)
    // При смене знака нижняя и верхняя границы меняются местами; L сохраняется.
    if (node.kind == "unary") return if (node.text == "-") Bound(-u.hi, -u.lo, u.lip) else u
    if (node.kind == "call") return when (node.text) {
        // |sin| и |cos| <= 1; модуль их производной также <= 1.
        // Поэтому для sin(u), cos(u) подходит константа L(u).
        "sin", "cos" -> Bound(-1.0, 1.0, u.lip)
        // exp возрастает; на диапазоне u её производная не больше exp(u.hi).
        "exp" -> Bound(exp(u.lo), exp(u.hi), exp(u.hi) * u.lip)
        // У log производная 1/u, у sqrt — 1/(2*sqrt(u)).
        // Положительная нижняя граница обеспечивает конечность этих оценок.
        // Если грубая оценка диапазона не подходит, автоматический расчёт прекращаем.
        "log" -> { require(u.lo > 0) { "Для log задайте корректное L вручную" }; Bound(ln(u.lo), ln(u.hi), u.lip/u.lo) }
        "sqrt" -> { require(u.lo > 0) { "Для sqrt задайте корректное L вручную" }; Bound(sqrt(u.lo), sqrt(u.hi), u.lip/(2*sqrt(u.lo))) }
        else -> Bound(if (u.lo <= 0 && u.hi >= 0) 0.0 else min(abs(u.lo),abs(u.hi)), max(abs(u.lo),abs(u.hi)),u.lip)
    }
    // Для бинарной операции отдельно оцениваем и правый операнд.
    val v = expressionBound(node.children[1], a, b)
    // m и n ограничивают |u| и |v| и нужны для произведения и частного.
    val m = max(abs(u.lo), abs(u.hi)); val n = max(abs(v.lo), abs(v.hi))
    return when (node.text) {
        // Для суммы и разности складываем константы Липшица.
        "+" -> Bound(u.lo+v.lo, u.hi+v.hi, u.lip+v.lip)
        "-" -> Bound(u.lo-v.hi, u.hi-v.lo, u.lip+v.lip)
        "*" -> {
            // Крайние значения произведения диапазонов ищем среди четырёх сочетаний концов.
            val values = listOf(u.lo*v.lo,u.lo*v.hi,u.hi*v.lo,u.hi*v.hi)
            // Для произведения: L(uv) <= max|v|*L(u) + max|u|*L(v).
            Bound(values.min(),values.max(),n*u.lip+m*v.lip)
        }
        "/" -> {
            // Для деления знаменатель должен быть отделён от нуля.
            require(v.lo > 0 || v.hi < 0) { "Знаменатель может быть нулевым; задайте L вручную" }
            val values = listOf(u.lo/v.lo,u.lo/v.hi,u.hi/v.lo,u.hi/v.hi)
            // d — нижняя граница |v|; L(u/v) <= L(u)/d + max|u|*L(v)/d^2.
            val d = min(abs(v.lo),abs(v.hi))
            Bound(values.min(),values.max(),u.lip/d+m*v.lip/(d*d))
        }
        else -> {
            // Автоматически обрабатываем только постоянный целый показатель.
            // Другие степени можно вычислять, но корректное L потребуется задать вручную.
            require(v.lip == 0.0 && v.lo == v.hi && v.lo == floor(v.lo)) { "Для этой степени задайте L вручную" }
            val p = v.lo
            when {
                p == 0.0 -> Bound(1.0,1.0,0.0)
                p > 0 -> {
                    val values = mutableListOf(u.lo.pow(p),u.hi.pow(p))
                    // При пересечении нуля добавляем 0 в диапазон.
                    // Для нечётной степени это может дать запас, но не сужает диапазон неверно.
                    if (u.lo <= 0 && u.hi >= 0) values.add(0.0)
                    // Из производной u^p: L <= p*max|u|^(p-1)*L(u).
                    Bound(values.min(),values.max(),p*m.pow(p-1)*u.lip)
                }
                else -> {
                    // Для отрицательной степени также исключаем ноль основания.
                    require(u.lo > 0 || u.hi < 0) { "Отрицательная степень при возможном нуле" }
                    val values = listOf(u.lo.pow(p),u.hi.pow(p))
                    Bound(values.min(),values.max(),abs(p)*min(abs(u.lo),abs(u.hi)).pow(p-1)*u.lip)
                }
            }
        }
    }
}

// Один интервал: его концы, нижняя оценка и следующая пробная точка.
// xl,xr — концы; fl,fr — значения на концах; candidate — координата пересечения.
// lower — высота этого пересечения, то есть минимум нижней ломаной на интервале.
data class Interval(val lower: Double, val xl: Double, val xr: Double,
                    val fl: Double, val fr: Double, val candidate: Double)
// Снимок состояния до новой пробы: номер шага, оценки, лучший x и следующий кандидат.
data class Step(val iteration: Int, val upper: Double, val lower: Double,
                val xBest: Double, val candidate: Double)
// Итог: входные данные, x и значение, нижняя/верхняя границы и их разрыв gap.
// iterations — число новых проб; evaluations включает ещё две начальные пробы.
// samples нужны для графика, history — для просмотра последовательности шагов.
data class Result(val expression: String, val a: Double, val b: Double, val eps: Double,
                  val lip: Double, val direction: String, val x: Double, val value: Double,
                  val lower: Double, val upper: Double, val gap: Double,
                  val iterations: Int, val evaluations: Int, val seconds: Double,
                  val samples: List<Pair<Double,Double>>, val history: List<Step>)

fun intersection(xl: Double, fl: Double, xr: Double, fr: Double, lip: Double): Interval {
    // Координата и высота пересечения двух опорных прямых.
    // Приравниваем fl-L*(x-xl) и fr-L*(xr-x).
    // Решение: середина интервала минус (fr-fl)/(2*L).
    val x = (xl+xr)/2 - (fr-fl)/(2*lip)
    // Подставляем найденный x в опорную прямую и получаем высоту eta.
    // Это значение ломаной, а не результат вычисления исходной функции в x.
    val lower = (fl+fr-lip*(xr-xl))/2
    return Interval(lower,xl,xr,fl,fr,x)
}

// expression — строка f(x), a,b — границы, eps — точность по значению функции.
// suppliedL — известная константа или null для автооценки; direction — min или max.
// maxIter ограничивает число новых проб и защищает от слишком долгого расчёта.
fun piyavskii(expression: String, a: Double, b: Double, eps: Double = 0.01,
              suppliedL: Double? = null, direction: String = "min", maxIter: Int = 100000): Result {
    // require сообщает об ошибочных аргументах; check ниже — о невозможности продолжать расчёт.
    // Проверяем отрезок, точность и направление поиска.
    require(a.isFinite() && b.isFinite() && a < b) { "Нужен конечный отрезок a < b" }
    require(eps.isFinite() && eps > 0 && maxIter > 0) { "eps и maxIter должны быть положительными" }
    require(direction in setOf("min", "max")) { "Направление: min или max" }
    // Строку разбираем один раз; при каждой пробе используем готовое дерево.
    val node = ExpressionParser(expression).parse()
    // Берём заданное L или оцениваем его по дереву выражения.
    // ?: выбирает правую часть только при null. Множитель 1+1e-12 даёт небольшой
    // численный запас; минимум 1e-12 предотвращает деление на ноль у константы.
    // Этот запас не заменяет математического обоснования L.
    val lip = suppliedL ?: max(expressionBound(node,a,b).lip*(1+1e-12),1e-12)
    require(lip.isFinite() && lip > 0) { "L должно быть конечным и положительным" }
    // Максимум ищем через минимум функции -f.
    val sign = if (direction == "min") 1.0 else -1.0
    // Для min работаем с f, для max — с -f; остальной алгоритм одинаков.
    // also проверяет конечность значения и возвращает то же вычисленное число.
    fun value(x: Double): Double = (sign*evaluate(node,x)).also {
        require(it.isFinite()) { "Функция должна быть конечной на отрезке" }
    }
    // Засекаем время самого метода, без разбора строки и графика.
    val started = System.nanoTime()
    // Две начальные пробы на концах отрезка.
    val fa = value(a); val fb = value(b)
    // Ключ — координата, значение — проба рабочей функции (f или -f).
    // TreeMap хранит ключи по возрастанию, поэтому точки для графика уже упорядочены.
    val samples = TreeMap<Double,Double>().apply { put(a,fa); put(b,fb) }
    // Лучшая проба даёт верхнюю оценку минимума.
    var xBest = if (fa <= fb) a else b
    // upper = U: наименьшее из уже вычисленных значений рабочей функции.
    var upper = min(fa,fb)
    // В начале очереди всегда интервал с самой низкой оценкой.
    // При одинаковом lower сравниваем xl, затем xr: выбор остаётся однозначным.
    // Очередь приоритетов быстро выдаёт минимум, но не сортирует все элементы целиком.
    val queue = PriorityQueue(compareBy<Interval> { it.lower }.thenBy { it.xl }.thenBy { it.xr })
    // История не влияет на выбор точек; сохраняем её для разбора хода метода.
    val history = mutableListOf<Step>()
    fun addInterval(xl: Double, fl: Double, xr: Double, fr: Double) {
        // Проверяем согласованность L с уже известными пробами.
        // Условие |fr-fl| <= L*(xr-xl) необходимо для известных концов.
        // Добавка 1e-10*масштаб допускает округление; проверка не доказывает L на всём отрезке.
        require(abs(fr-fl) <= lip*(xr-xl)+1e-10*maxOf(1.0,abs(fl),abs(fr))) { "L противоречит известным пробам" }
        // Создаём нижнюю вершину аналитически и помещаем интервал в очередь.
        queue.add(intersection(xl,fl,xr,fr,lip))
    }
    // Вначале известен один интервал между двумя начальными пробами.
    addInterval(a,fa,b,fb)
    // Счётчик пока равен нулю: пробы на концах не считаем итерациями.
    var iteration = 0
    while (true) {
        // 1. Выбираем самый перспективный интервал.
        // peek читает лучший интервал без удаления; current.lower — глобальная нижняя оценка.
        val current = queue.peek()
        // Сохраняем состояние до проверки остановки, включая последнее состояние.
        history.add(Step(iteration,upper,current.lower,xBest,current.candidate))
        // 2. Останавливаемся, когда разрыв оценок не больше eps.
        // lower <= точный минимум <= upper, поэтому upper-lower ограничивает ошибку
        // значения функции. Это не оценка расстояния от xBest до точного аргумента.
        if (upper-current.lower <= eps) break
        // При исчерпании лимита выдаём ошибку: достигнутая точность ещё не подтверждена.
        check(iteration < maxIter) { "Превышен maxIter; точность не достигнута" }
        // 3. Удаляем интервал и вычисляем функцию в его нижней вершине.
        // После добавления пробы старого интервала больше не будет.
        queue.remove()
        // Новая координата — пересечение прямых, а не точка равномерной сетки.
        val x = current.candidate
        // Из-за округления координата может совпасть с концом или старой пробой.
        // В таком случае прекращаем деление, чтобы не зациклиться.
        check(x > current.xl && x < current.xr && !samples.containsKey(x)) { "Предел машинной точности" }
        // Единственное новое вычисление рабочей функции за эту итерацию.
        val fx = value(x)
        // Сохраняем верхнюю вершину: теперь реальное значение в этой точке известно.
        samples[x] = fx
        // 4. При необходимости обновляем лучший найденный ответ.
        // Меняем одновременно координату лучшей пробы и её значение; при равенстве сохраняем старую.
        if (fx < upper) { xBest = x; upper = fx }
        // 5. Вместо выбранного интервала добавляем два новых.
        // Слева остаётся старый левый конец, справа концом становится новая проба.
        addInterval(current.xl,current.fl,x,fx)
        // Во втором интервале новая проба — левый конец; прежний правый сохраняется.
        addInterval(x,fx,current.xr,current.fr)
        // Один завершённый шаг добавил ровно одну пробу; всего проб iteration+2.
        iteration++
    }
    // Разницу наносекунд переводим в секунды; график будет построен позже.
    val elapsed = (System.nanoTime()-started)/1e9
    // После остановки очередь уже содержит вершины окончательной ломаной.
    val lower = queue.peek().lower
    // Возвращаем ответ и границы в знаках исходной функции.
    // Для max: из lower <= min(-f) <= upper получаем -upper <= max(f) <= -lower.
    // Разрыв не меняется; значения сохранённых проб также переводим обратно к f.
    return Result(expression,a,b,eps,lip,direction,xBest,sign*upper,
        if (sign == 1.0) lower else -upper, if (sign == 1.0) upper else -lower,
        upper-lower,iteration,samples.size,elapsed,samples.map { it.key to sign*it.value },history)
}

fun plotResult(result: Result, path: String) {
    // Сетка нужна только для рисунка: оптимизация выше её не использует.
    val node = ExpressionParser(result.expression).parse()
    // Для max пересечения сначала строим для -f, затем отражаем обратно для рисунка.
    val sign = if (result.direction == "min") 1.0 else -1.0
    // Восстанавливаем ломаную по соседним пробам и их пересечениям.
    val polygon = mutableListOf(result.samples.first())
    // zipWithNext перебирает только соседние пробы, которые ограничивают один интервал.
    result.samples.zipWithNext().forEach { (left,right) ->
        val v = intersection(left.first,sign*left.second,right.first,sign*right.second,result.lip)
        // Порядок точек ломаной: проба, нижняя вершина, следующая проба.
        polygon.add(v.candidate to sign*v.lower); polygon.add(right)
    }
    // 2501 равномерная точка нужна только для плавной линии исходного графика.
    // К этому моменту минимум и его оценки уже найдены.
    val curve = (0..2500).map { i ->
        val x = result.a+(result.b-result.a)*i/2500.0
        x to evaluate(node,x)
    }
    // Создаём изображение с двумя панелями: весь отрезок и увеличение возле ответа.
    val image = BufferedImage(1600,650,BufferedImage.TYPE_INT_RGB)
    val g = image.createGraphics()
    g.setRenderingHint(RenderingHints.KEY_ANTIALIASING,RenderingHints.VALUE_ANTIALIAS_ON)
    g.color = Color.WHITE; g.fillRect(0,0,image.width,image.height)
    fun panel(left: Int, xmin: Double, xmax: Double, ymin: Double, ymax: Double, title: String) {
        val top = 70; val width = 650; val height = 440
        // Переводим координаты графика в пиксели изображения.
        fun px(x: Double) = left+((x-xmin)/(xmax-xmin)*width).toInt()
        // На изображении y растёт вниз, поэтому вертикальную координату переворачиваем.
        fun py(y: Double) = top+height-((y-ymin)/(ymax-ymin)*height).toInt()
        g.font = Font("SansSerif",Font.PLAIN,18)
        for (i in 0..5) {
            val x = xmin+(xmax-xmin)*i/5; val y = ymin+(ymax-ymin)*i/5
            g.color = Color(225,230,235); g.drawLine(px(x),top,px(x),top+height);g.drawLine(left,py(y),left+width,py(y))
            g.color = Color.DARK_GRAY
            g.drawString("%.2f".format(Locale.US,x),px(x)-23,top+height+30)
            g.drawString("%.2f".format(Locale.US,y),left-65,py(y)+5)
        }
        g.color = Color.DARK_GRAY; g.drawRect(left,top,width,height)
        g.font = Font("SansSerif",Font.BOLD,21);g.drawString(title,left,38)
        // Ограничиваем линии рамкой панели: части графика вне диапазона не рисуются.
        val oldClip = g.clip
        g.clipRect(left,top,width,height)
        fun line(points: List<Pair<Double,Double>>, color: Color, thickness: Float) {
            g.color = color; g.stroke = BasicStroke(thickness)
            points.zipWithNext().forEach { (u,v) -> g.drawLine(px(u.first),py(u.second),px(v.first),py(v.second)) }
        }
        // Синяя линия — функция, оранжевая — вспомогательная ломаная.
        line(curve,Color(29,79,145),2.5f);line(polygon,Color(212,122,20),1.4f)
        g.color = Color(120,135,150)
        // Серые точки — все пробы метода; красная — лучший найденный результат.
        result.samples.forEach { (x,y) -> g.fillOval(px(x)-2,py(y)-2,4,4) }
        g.color = Color(187,34,51);g.fillOval(px(result.x)-7,py(result.value)-7,14,14)
        g.clip = oldClip
        g.font = Font("SansSerif",Font.PLAIN,18)
        g.color = Color(29,79,145);g.drawString("Синий: f(x)",left,610)
        g.color = Color(212,122,20);g.drawString("Оранжевый: ломаная",left+155,610)
        g.color = Color(187,34,51);g.drawString("Красный: экстремум",left+390,610)
        g.color = Color.DARK_GRAY;g.drawString("x",left+width/2,top+height+58);g.drawString("f(x)",left-65,top-15)
    }
    // Масштаб общей панели учитывает и функцию, и нижнюю ломаную.
    val ys = (curve+polygon).map { it.second }
    val padding = max(1.0,(ys.max()-ys.min())*0.05)
    panel(85,result.a,result.b,ys.min()-padding,ys.max()+padding,"Kotlin: весь отрезок")
    // Вторая панель показывает небольшой участок вокруг найденного x.
    val zoom = min((result.b-result.a)/4,0.15)
    panel(885,max(result.a,result.x-zoom),min(result.b,result.x+zoom),
          result.value-max(0.03,3*result.eps),result.value+max(0.1,10*result.eps),"Увеличение около экстремума")
    // Освобождаем графический контекст, создаём каталог и сохраняем PNG.
    g.dispose()
    File(path).absoluteFile.parentFile.mkdirs()
    ImageIO.write(image,"png",File(path))
}

// Проверяем метод на функциях с известными ответами.
fun selfTest() {
    data class Case(val f: String,val a: Double,val b: Double,val mode: String,val exact: Double)
    val cases = listOf(Case("(x-0.37)**2",-1.0,2.0,"min",0.0),Case("x",-2.0,3.0,"min",-2.0),
        Case("5",-1.0,1.0,"min",5.0),Case("-(x-0.2)**2",-1.0,1.0,"max",0.0),Case("abs(x-0.3)",-1.0,1.0,"min",0.0))
    for (c in cases) {
        val r = piyavskii(c.f,c.a,c.b,1e-4,direction=c.mode)
        // Проверяем попадание точного значения в границы, ошибку ответа и достигнутый gap.
        check(r.lower <= c.exact+1e-9 && c.exact <= r.upper+1e-9 && abs(r.value-c.exact)<=1e-4 && r.gap<=1e-4)
        println("OK: ${c.f}, ${c.mode}, iterations=${r.iterations}")
    }
    // Отдельно проверяем приоритет знака и правую ассоциативность степени.
    check(evaluate(ExpressionParser("-x**2").parse(),2.0) == -4.0)
    check(evaluate(ExpressionParser("2**3**2").parse(),0.0) == 512.0)
}

fun main(args: Array<String>) {
    // В числовом выводе используем точку как десятичный разделитель.
    Locale.setDefault(Locale.US)
    // Специальный аргумент запускает самопроверки вместо демонстрации.
    if (args.firstOrNull() == "--self-test") { selfTest(); return }
    require(args.isEmpty() || args.size in 4..7) { "Аргументы: expression a b eps [min|max] [L|auto] [png]" }
    // Читаем параметры; без аргументов запускаем пример Растригина.
    // getOrNull не выходит за границы массива; ?: подставляет значение по умолчанию.
    val expression = args.getOrNull(0) ?: "x**2 + 10*(1-cos(2*pi*x))"
    // Следующие три аргумента задают левую границу, правую границу и требуемый gap.
    val a = args.getOrNull(1)?.toDouble() ?: -2.0
    val b = args.getOrNull(2)?.toDouble() ?: 2.0
    val eps = args.getOrNull(3)?.toDouble() ?: 0.01
    // Необязательное направление; при отсутствии ищем минимум.
    val mode = args.getOrNull(4) ?: "min"
    // Слово auto означает отсутствие заданного L: в метод передаётся null.
    val suppliedL = args.getOrNull(5)?.takeUnless { it == "auto" }?.toDouble()
    // Последний необязательный аргумент — путь сохраняемого графика.
    val path = args.getOrNull(6) ?: "kotlin-visualization.png"
    // Выполняем расчёт, выводим ответ и сохраняем график.
    val r = piyavskii(expression,a,b,eps,suppliedL,mode)
    println("expression=$expression")
    println("x=${r.x}\nf=${r.value}\nL=${r.lip}\nlower=${r.lower}\nupper=${r.upper}\ngap=${r.gap}")
    // evaluations учитывает концы; seconds — время оптимизации без рисования.
    println("iterations=${r.iterations}\nevaluations=${r.evaluations}\nseconds=${r.seconds}")
    plotResult(r,path)
    println("plot=$path")
}
